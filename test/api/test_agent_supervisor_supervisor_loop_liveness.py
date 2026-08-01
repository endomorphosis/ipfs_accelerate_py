"""Focused watchdog liveness tests for zero-write event-driven daemons."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import ManagedDaemonSpec
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import (
    SupervisorLoop,
    SupervisorLoopConfig,
    SupervisorLoopDecision,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    SupervisedChild,
    SupervisedChildSpec,
    adopt_supervised_child,
)


def _loop(tmp_path: Path, *, stale_after_seconds: float = 60.0) -> tuple[SupervisorLoop, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    status_path = state_dir / "daemon_status.json"
    spec = ManagedDaemonSpec(
        name="event-driven-daemon",
        schema="test.event-driven-daemon",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=(sys.executable, "-c", "pass"),
        status_path=status_path,
        supervisor_status_path=state_dir / "supervisor_status.json",
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure_status.json",
        ensure_check_path=state_dir / "ensure_check.json",
        latest_log_path=state_dir / "latest.log",
    )
    return (
        SupervisorLoop(
            SupervisorLoopConfig(
                spec=spec,
                command=(sys.executable, "-c", "pass"),
                log_prefix="child",
                watchdog_stale_after_seconds=stale_after_seconds,
            )
        ),
        status_path,
    )


def _child(log_path: Path) -> SupervisedChild:
    return SupervisedChild(
        pid=os.getpid(),
        command=(sys.executable, "-c", "pass"),
        log_path=log_path,
        child_pid_path=log_path.parent / "child.pid",
    )


def _timestamp(seconds_ago: float) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    ).isoformat()


def test_supervisor_loop_uses_fresh_child_log_when_canonical_heartbeat_is_stale(
    tmp_path: Path,
) -> None:
    loop, status_path = _loop(tmp_path)
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps({"heartbeat_at": _timestamp(600)}),
        encoding="utf-8",
    )
    original_bytes = status_path.read_bytes()
    original_mtime_ns = status_path.stat().st_mtime_ns
    log_path = status_path.parent / "child.log"
    log_path.write_text("unchanged pass complete\n", encoding="utf-8")

    decision = loop.default_watchdog(
        _child(log_path),
        json.loads(status_path.read_text(encoding="utf-8")),
    )

    assert decision.action == "continue"
    assert loop._last_liveness_status["canonical_state"]["stale"] is True
    assert loop._last_liveness_status["child_log"]["fresh"] is True
    assert loop._last_liveness_status["effective_heartbeat_source"] == "child_log"
    assert loop._last_liveness_status["effective_heartbeat_fresh"] is True
    assert status_path.read_bytes() == original_bytes
    assert status_path.stat().st_mtime_ns == original_mtime_ns


def test_supervisor_loop_uses_freshest_canonical_or_child_log_signal(
    tmp_path: Path,
) -> None:
    loop, status_path = _loop(tmp_path)
    log_path = status_path.parent / "child.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("old output\n", encoding="utf-8")
    old_log_at = time.time() - 120
    os.utime(log_path, (old_log_at, old_log_at))

    decision = loop.default_watchdog(
        _child(log_path),
        {"heartbeat_at": _timestamp(1)},
    )

    assert decision.action == "continue"
    assert loop._last_liveness_status["child_log"]["stale"] is True
    assert loop._last_liveness_status["canonical_state"]["fresh"] is True
    assert loop._last_liveness_status["effective_heartbeat_source"] == "canonical_state"


def test_supervisor_loop_recycles_when_canonical_and_child_log_are_stale(
    tmp_path: Path,
) -> None:
    loop, status_path = _loop(tmp_path)
    log_path = status_path.parent / "child.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("old output\n", encoding="utf-8")
    old_log_at = time.time() - 120
    os.utime(log_path, (old_log_at, old_log_at))

    decision = loop.default_watchdog(
        _child(log_path),
        {"heartbeat_at": _timestamp(180)},
    )

    assert decision.action == "recycle"
    assert decision.reason == "stale_heartbeat"
    assert decision.detail["canonical_state"]["stale"] is True
    assert decision.detail["child_log"]["stale"] is True
    assert decision.detail["effective_heartbeat_source"] == "child_log"
    assert decision.detail["all_signals_stale_or_missing"] is True


def test_supervisor_loop_recycles_when_all_liveness_signals_are_missing(
    tmp_path: Path,
) -> None:
    loop, status_path = _loop(tmp_path)

    decision = loop.default_watchdog(
        _child(status_path.parent / "missing.log"),
        {},
    )

    assert decision.action == "recycle"
    assert decision.reason == "stale_heartbeat"
    assert decision.detail["canonical_state"]["missing"] is True
    assert decision.detail["child_log"]["missing"] is True
    assert decision.detail["effective_heartbeat_source"] == "missing"
    assert decision.detail["effective_heartbeat_age_seconds"] is None


def test_supervisor_loop_publishes_effective_liveness_diagnostics(
    tmp_path: Path,
) -> None:
    loop, status_path = _loop(tmp_path)
    log_path = status_path.parent / "child.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("pass complete\n", encoding="utf-8")
    child = _child(log_path)

    assert loop.default_watchdog(child, {"heartbeat_at": _timestamp(600)}).action == "continue"
    loop._write_status("running", child=child)

    supervisor_status = json.loads(
        loop.config.spec.supervisor_status_path.read_text(encoding="utf-8")
    )
    diagnostics = supervisor_status["watchdog_liveness"]
    assert diagnostics["effective_heartbeat_source"] == "child_log"
    assert diagnostics["effective_heartbeat_fresh"] is True
    assert diagnostics["canonical_state"]["stale"] is True


def test_stale_default_watchdog_does_not_invoke_custom_hook(
    tmp_path: Path,
) -> None:
    loop, status_path = _loop(tmp_path)
    hook_calls: list[int] = []

    def mutating_hook(
        _loop: SupervisorLoop,
        child: SupervisedChild,
        _status: dict[str, object],
    ) -> SupervisorLoopDecision:
        hook_calls.append(child.pid)
        return SupervisorLoopDecision.stop("maintenance hook invoked")

    loop.watchdog_hook = mutating_hook

    decision = loop.watchdog_decision(
        _child(status_path.parent / "missing.log"),
    )

    assert decision.action == "recycle"
    assert decision.reason == "stale_heartbeat"
    assert hook_calls == []


def test_fresh_default_watchdog_still_invokes_custom_hook(
    tmp_path: Path,
) -> None:
    loop, status_path = _loop(tmp_path)
    log_path = status_path.parent / "child.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("fresh output\n", encoding="utf-8")
    hook_calls: list[int] = []

    def hook(
        _loop: SupervisorLoop,
        child: SupervisedChild,
        _status: dict[str, object],
    ) -> SupervisorLoopDecision:
        hook_calls.append(child.pid)
        return SupervisorLoopDecision.stop("requested by hook")

    loop.watchdog_hook = hook

    decision = loop.watchdog_decision(_child(log_path))

    assert decision.action == "stop"
    assert decision.reason == "requested by hook"
    assert hook_calls == [os.getpid()]


def test_supervisor_liveness_ignores_retargeted_latest_log_alias(
    tmp_path: Path,
) -> None:
    loop, status_path = _loop(tmp_path)
    logs = status_path.parent
    run_log = logs / "child-run.log"
    run_log.parent.mkdir(parents=True)
    run_log.write_text("old run output\n", encoding="utf-8")
    old_log_at = time.time() - 120
    os.utime(run_log, (old_log_at, old_log_at))
    unrelated_log = logs / "unrelated-fresh.log"
    unrelated_log.write_text("fresh unrelated output\n", encoding="utf-8")
    latest_log = logs / "latest.log"
    latest_log.symlink_to(unrelated_log.name)
    child = SupervisedChild(
        pid=os.getpid(),
        command=(sys.executable, "-c", "pass"),
        log_path=run_log,
        child_pid_path=logs / "child.pid",
        latest_log_path=latest_log,
    )

    decision = loop.default_watchdog(
        child,
        {"heartbeat_at": _timestamp(180)},
    )

    assert decision.action == "recycle"
    assert decision.reason == "stale_heartbeat"
    assert decision.detail["child_log"]["path"] == str(run_log)
    assert decision.detail["child_log"]["stale"] is True


def test_adopted_child_binds_liveness_to_process_stdout_not_latest_alias(
    tmp_path: Path,
) -> None:
    loop, status_path = _loop(tmp_path)
    repo = loop.config.spec.repo_root
    logs = repo / "logs"
    state = repo / "state"
    logs.mkdir(parents=True)
    state.mkdir(parents=True, exist_ok=True)
    actual_log = logs / "actual-run.log"
    command = ("/bin/sleep", "30")
    with actual_log.open("ab") as out_handle:
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdin=subprocess.DEVNULL,
            stdout=out_handle,
            stderr=subprocess.STDOUT,
        )
    try:
        stdout_fd = Path("/proc") / str(process.pid) / "fd" / "1"
        if not stdout_fd.exists():
            pytest.skip("procfs stdout descriptors are required for this regression")
        assert process.poll() is None

        pid_path = state / "child.pid"
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
        first_decoy = logs / "first-decoy.log"
        first_decoy.write_text("unrelated output\n", encoding="utf-8")
        latest_log = logs / "latest.log"
        latest_log.symlink_to(first_decoy.name)
        child = adopt_supervised_child(
            SupervisedChildSpec(
                repo_root=repo,
                command=command,
                log_path=Path("logs/proposed-new-run.log"),
                child_pid_path=Path("state/child.pid"),
                latest_log_path=Path("logs/latest.log"),
            )
        )

        assert child is not None
        assert child.log_path == actual_log.resolve()
        assert child.log_path != latest_log.resolve()

        second_decoy = logs / "second-decoy.log"
        second_decoy.write_text("fresh unrelated output\n", encoding="utf-8")
        latest_log.unlink()
        latest_log.symlink_to(second_decoy.name)
        old_log_at = time.time() - 120
        os.utime(actual_log, (old_log_at, old_log_at))

        decision = loop.default_watchdog(
            child,
            {"heartbeat_at": _timestamp(180)},
        )

        assert decision.action == "recycle"
        assert decision.detail["child_log"]["path"] == str(actual_log.resolve())
        assert decision.detail["child_log"]["stale"] is True
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
