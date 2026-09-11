"""Exact disposable process recovery; no native board is touched."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    native_graceful_recovery as recovery,
)

FIXTURE = Path(__file__).parent / "fixtures/sawm_graceful_process_fixture.py"


def wait(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("disposable fixture did not reach expected state")


@contextlib.contextmanager
def actors(directory, mode="normal"):
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", str(FIXTURE), "master", str(directory), mode],
        start_new_session=True,
    )
    roster = {}
    try:
        wait(
            lambda: (
                (directory / "roster.json").exists()
                and (directory / "daemon-ready").exists()
            )
        )
        roster = json.loads((directory / "roster.json").read_text())
        controller = recovery.observe_process(roster["controller"])
        lane = recovery.LaneBinding(
            recovery.observe_process(roster["supervisor"]),
            recovery.observe_process(roster["daemon"]),
        )
        yield controller, [lane]
    finally:
        # Test teardown only: owned fixture actors, never production PIDs.
        # Stop master first so it cannot create new fixture children.
        if process.poll() is None:
            os.kill(process.pid, signal.SIGSTOP)
        pids = list(roster.values())
        if (directory / "births").exists():
            pids.extend(int(v) for v in (directory / "births").read_text().splitlines())
        if (directory / "ancillary.pid").exists():
            pids.append(int((directory / "ancillary.pid").read_text()))
        for pid in dict.fromkeys(reversed(pids)):
            if pid == process.pid:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def options(directory, controller, lanes):
    @contextlib.contextmanager
    def fence(index):
        assert index == 0
        with (directory / ".lane.lock.update.lock").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def closed():
        assert (directory / "supervisor-cleanup").exists()

    return dict(  # noqa: C408 - keyword-shaped mechanism arguments
        controller=controller,
        lanes=lanes,
        lane_fence=fence,
        effect_gate=lambda: None,
        lane_children_gate=lambda _: None,
        closed_children_gate=closed,
        record_phase=lambda _: None,
        timeout_seconds=2,
    )


def test_real_native_handler_prevents_relaunch_and_all_master_threads_stop(
    tmp_path, monkeypatch
):
    with actors(tmp_path) as (controller, lanes):
        seen = []
        native = signal.pidfd_send_signal

        def signal_spy(fd, sig, *args):
            seen.append(sig)
            return native(fd, sig, *args)

        monkeypatch.setattr(signal, "pidfd_send_signal", signal_spy)
        kwargs = options(tmp_path, controller, lanes)
        phases = []

        def record(phase):
            phases.append(phase)
            if phase == "controller_all_threads_stopped":
                assert len(list(Path(f"/proc/{controller.pid}/task").iterdir())) >= 2
                assert recovery.all_threads_stopped(controller)

        kwargs["record_phase"] = record
        result = recovery.gracefully_close_native_lanes(**kwargs)
        assert result["controller_exited"] is True
        assert result["callback_settlement_authority"] is False
        assert seen == [
            signal.SIGSTOP,
            signal.SIGSTOP,
            signal.SIGTERM,
            signal.SIGTERM,
            signal.SIGCONT,
            signal.SIGTERM,
            signal.SIGCONT,
        ]
        assert (tmp_path / "master-cleanup").exists()
        assert len((tmp_path / "births").read_text().splitlines()) == 1
        assert phases[-1] == "controller_exit_observed"


@pytest.mark.parametrize(
    "phase",
    [
        "controller_suspend_prepared",
        "controller_all_threads_stopped",
        "supervisor_suspend_prepared",
        "supervisor_all_threads_stopped",
        "daemon_graceful_exit_prepared",
        "daemon_exit_observed",
        "lane_exit_observed",
        "controller_graceful_exit_prepared",
    ],
)
def test_every_preterminal_journal_failure_resumes_exact_master(tmp_path, phase):
    with actors(tmp_path) as (controller, lanes):
        kwargs = options(tmp_path, controller, lanes)

        def fail(value):
            if value == phase:
                raise OSError("disposable journal fault")

        kwargs["record_phase"] = fail
        with pytest.raises(OSError, match="disposable journal"):
            recovery.gracefully_close_native_lanes(**kwargs)
        wait(lambda: recovery._stat(controller.pid)[0] not in {"T", "t"})


def test_real_late_daemon_child_denies_wrapper_forced_cleanup(tmp_path, monkeypatch):
    with actors(tmp_path, "ancillary") as (controller, lanes):
        kwargs = options(tmp_path, controller, lanes)

        def census(_):
            pid = int((tmp_path / "ancillary.pid").read_text())
            assert recovery._stat(pid)[0] not in {"Z", "X"}
            raise recovery.GracefulRecoveryUnverified("late_fixture_child")

        kwargs["lane_children_gate"] = census
        with pytest.raises(
            recovery.GracefulRecoveryUnverified, match="late_fixture_child"
        ):
            recovery.gracefully_close_native_lanes(**kwargs)
        assert not (tmp_path / "supervisor-cleanup").exists()
        assert not (tmp_path / "master-cleanup").exists()
        wait(lambda: recovery._stat(controller.pid)[0] not in {"T", "t"})


@pytest.mark.parametrize("ordinal", [1, 2, 3, 4, 6])
def test_signal_failures_attempt_resume_without_escalation(
    tmp_path, monkeypatch, ordinal
):
    with actors(tmp_path) as (controller, lanes):
        kwargs = options(tmp_path, controller, lanes)
        seen = []
        native = signal.pidfd_send_signal

        def fault(fd, sig, *args):
            seen.append(sig)
            if len(seen) == ordinal:
                raise PermissionError("disposable signal denial")
            return native(fd, sig, *args)

        monkeypatch.setattr(signal, "pidfd_send_signal", fault)
        with pytest.raises(PermissionError, match="disposable signal"):
            recovery.gracefully_close_native_lanes(**kwargs)
        assert signal.SIGKILL not in seen
        assert seen[-1] == signal.SIGCONT
        wait(lambda: recovery._stat(controller.pid)[0] not in {"T", "t"})


def test_existing_operator_suspension_is_not_resumed(tmp_path, monkeypatch):
    with actors(tmp_path) as (controller, lanes):
        os.kill(controller.pid, signal.SIGSTOP)
        wait(lambda: recovery.all_threads_stopped(controller))
        monkeypatch.setattr(
            signal, "pidfd_send_signal", lambda *_: pytest.fail("unexpected signal")
        )
        with pytest.raises(
            recovery.GracefulRecoveryUnverified, match="already_stopped"
        ):
            recovery.gracefully_close_native_lanes(
                **options(tmp_path, controller, lanes)
            )
        assert recovery.all_threads_stopped(controller)
        os.kill(controller.pid, signal.SIGCONT)
        recovery.require_exact_process(controller)


def test_ignored_daemon_term_times_out_without_escalation_and_resumes_master(
    tmp_path, monkeypatch
):
    with actors(tmp_path, "ignore") as (controller, lanes):
        kwargs = options(tmp_path, controller, lanes)
        kwargs["timeout_seconds"] = 0.15
        seen = []
        native = signal.pidfd_send_signal

        def signal_spy(fd, sig, *args):
            seen.append(sig)
            return native(fd, sig, *args)

        monkeypatch.setattr(signal, "pidfd_send_signal", signal_spy)
        with pytest.raises(recovery.GracefulRecoveryUnverified, match="no_escalation"):
            recovery.gracefully_close_native_lanes(**kwargs)
        assert seen == [
            signal.SIGSTOP,
            signal.SIGSTOP,
            signal.SIGTERM,
            signal.SIGCONT,
            signal.SIGCONT,
        ]
        recovery.require_exact_process(lanes[0].daemon)
        wait(lambda: recovery._stat(controller.pid)[0] not in {"T", "t"})


def test_wrong_birth_refuses_before_any_signal(tmp_path, monkeypatch):
    with actors(tmp_path) as (controller, lanes):
        kwargs = options(
            tmp_path, replace(controller, birth=controller.birth + 1), lanes
        )
        monkeypatch.setattr(
            signal, "pidfd_send_signal", lambda *_: pytest.fail("unexpected signal")
        )
        with pytest.raises(
            recovery.GracefulRecoveryUnverified, match="binding_changed"
        ):
            recovery.gracefully_close_native_lanes(**kwargs)


@pytest.mark.parametrize("which", ["lane", "final"])
def test_remaining_child_census_denial_never_signals_parent(
    tmp_path, which, monkeypatch
):
    with actors(tmp_path) as (controller, lanes):
        kwargs = options(tmp_path, controller, lanes)

        def deny(*_):
            raise recovery.GracefulRecoveryUnverified("remaining_fixture_child")

        kwargs["lane_children_gate" if which == "lane" else "closed_children_gate"] = (
            deny
        )
        seen = []
        native = signal.pidfd_send_signal

        def signal_spy(fd, sig, *args):
            seen.append(sig)
            return native(fd, sig, *args)

        monkeypatch.setattr(signal, "pidfd_send_signal", signal_spy)
        with pytest.raises(
            recovery.GracefulRecoveryUnverified, match="remaining_fixture_child"
        ):
            recovery.gracefully_close_native_lanes(**kwargs)
        assert seen.count(signal.SIGTERM) == (1 if which == "lane" else 2)
        wait(lambda: recovery._stat(controller.pid)[0] not in {"T", "t"})
