"""Exclusive owners survive session SIGTERM after identity is observed."""

from __future__ import annotations

import os
import signal
import sys
import textwrap
import threading
from pathlib import Path

from scripts import run_agent_supervisor_efficiency_state_hardening as aseh_operator

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    build_arg_parser,
    build_configured_multi_supervisor_cli_runner,
    parse_track_spec,
    run_supervisor_tracks,
)


def _sleeping_worker(path: Path, *, exit_on_sigterm: bool) -> None:
    handler = "sys.exit(0)" if exit_on_sigterm else "None"
    path.write_text(
        textwrap.dedent(
            f"""
            import signal
            import sys
            import time
            signal.signal(signal.SIGTERM, lambda *_: {handler})
            print("worker started", flush=True)
            while True:
                time.sleep(0.05)
            """
        ).lstrip(),
        encoding="utf-8",
    )


def test_cli_parser_accepts_survive_external_sigterm() -> None:
    args = build_arg_parser().parse_args(
        ["--track", "T|a.py|l.log|s.pid|d.pid", "--survive-external-sigterm"]
    )
    assert args.survive_external_sigterm is True
    default = build_arg_parser().parse_args(
        ["--track", "T|a.py|l.log|s.pid|d.pid"]
    )
    assert default.survive_external_sigterm is False


def test_infinite_duration_runner_argv_includes_survive_flag(tmp_path: Path) -> None:
    runner = build_configured_multi_supervisor_cli_runner(
        repo_root=tmp_path,
        duration_seconds=float("inf"),
        survive_external_sigterm=True,
        tracks=["T|a.py|l.log|s.pid|d.pid"],
    )
    assert "--survive-external-sigterm" in runner.args()
    finite = build_configured_multi_supervisor_cli_runner(
        repo_root=tmp_path,
        duration_seconds=1,
        tracks=["T|a.py|l.log|s.pid|d.pid"],
    )
    assert "--survive-external-sigterm" not in finite.args()


def test_run_supervisor_tracks_survives_sigterm_when_flag_set(tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    _sleeping_worker(worker, exit_on_sigterm=False)
    track = parse_track_spec(
        "T|worker.py|logs/{stamp}.log|state/supervisor.pid|state/daemon.pid",
        stamp="SIGTERM-SURVIVE",
    )
    output: list[str] = []
    sent = {"n": 0}

    def emit(message: str) -> None:
        output.append(message)
        if sent["n"] == 0 and "started T supervisor" in message:
            sent["n"] += 1
            os.kill(os.getpid(), signal.SIGTERM)

    result = run_supervisor_tracks(
        [track],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.25,
        heartbeat_interval_seconds=0.05,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        master_pid_path=tmp_path / "state" / "master.pid",
        label="sigterm-survive",
        survive_external_sigterm=True,
        output=emit,
    )
    assert result["completed"] is True
    assert result["interrupted"] == ""
    assert any(
        "ignored external SIGTERM after exclusive-owner identity" in line
        for line in output
    )


def test_run_supervisor_tracks_still_stops_on_sigterm_by_default(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "worker.py"
    _sleeping_worker(worker, exit_on_sigterm=True)
    track = parse_track_spec(
        "T|worker.py|logs/{stamp}.log|state/supervisor.pid|state/daemon.pid",
        stamp="SIGTERM-DEFAULT",
    )
    output: list[str] = []
    sent = {"n": 0}

    def emit(message: str) -> None:
        output.append(message)
        if sent["n"] == 0 and "started T supervisor" in message:
            sent["n"] += 1
            os.kill(os.getpid(), signal.SIGTERM)

    result = run_supervisor_tracks(
        [track],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=2.0,
        heartbeat_interval_seconds=0.05,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        master_pid_path=tmp_path / "state" / "master.pid",
        label="sigterm-default",
        output=emit,
    )
    assert result["completed"] is False
    assert "received signal 15" in str(result["interrupted"])


def test_stop_signal_handlers_default_still_catch_sigterm() -> None:
    requested = threading.Event()
    received: dict[str, int] = {}
    prior = signal.getsignal(signal.SIGTERM)
    with aseh_operator._stop_signal_handlers(requested, received):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        assert requested.is_set()
        assert received == {"signum": signal.SIGTERM}
    assert signal.getsignal(signal.SIGTERM) == prior


def test_stop_signal_handlers_survive_external_sigterm_ignores_sigterm() -> None:
    requested = threading.Event()
    received: dict[str, int] = {}
    prior = signal.getsignal(signal.SIGTERM)
    with aseh_operator._stop_signal_handlers(
        requested,
        received,
        survive_external_sigterm=True,
    ):
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert requested.is_set()
        assert received == {"signum": signal.SIGINT}
    assert signal.getsignal(signal.SIGTERM) == prior


def test_post_admission_continues_parallel_scope_when_recovery_is_not_admitted() -> None:
    scoped = {
        "healthy": False,
        "blocked": True,
        "stuck": False,
        "terminal": False,
        "scheduler_alive": True,
        "owner_ready": True,
        "broker_ready": True,
        "blocked_recovery_admitted": False,
        "blocked_recovery_scope": "parallel_startup",
    }
    assert aseh_operator._post_admission_health_action(
        scoped,
        prior_available=True,
        current_available=True,
        unhealthy_edges=2,
    ) == ("continue", "", 0)
    working = {**scoped, "blocked_recovery_scope": "parallel_work"}
    assert aseh_operator._post_admission_health_action(
        working,
        prior_available=True,
        current_available=True,
        unhealthy_edges=0,
    )[:2] == ("continue", "")
    halted = {**scoped, "blocked_recovery_scope": ""}
    assert aseh_operator._post_admission_health_action(
        halted,
        prior_available=True,
        current_available=True,
        unhealthy_edges=0,
    )[:2] == ("fail", "authoritative_board_blocked")
    mutated = {
        **scoped,
        "blocked_recovery_scope": "parallel_work",
        "source_identity_admitted": False,
    }
    assert aseh_operator._post_admission_health_action(
        mutated,
        prior_available=True,
        current_available=True,
        unhealthy_edges=0,
    )[:2] == ("fail", "authoritative_board_blocked")


def test_call_stop_signal_handlers_accepts_two_argument_test_double() -> None:
    from contextlib import nullcontext

    original = aseh_operator._stop_signal_handlers
    aseh_operator._stop_signal_handlers = lambda *_args: nullcontext()
    try:
        with aseh_operator._call_stop_signal_handlers(
            threading.Event(),
            {},
            survive_external_sigterm=True,
        ):
            pass
    finally:
        aseh_operator._stop_signal_handlers = original
