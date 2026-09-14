"""Real isolated native-handler roster; no board database or live actor."""

import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    native_phased_graceful_recovery as phased,
)

p = phased.process
FIXTURE = Path(__file__).parent / "fixtures/sawm_phased_process_fixture.py"


class PopulationPresent(RuntimeError):
    pass


def wait(predicate, seconds=5):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("fixture state timeout")


def alive(pid):
    try:
        return p._stat(pid)[0] not in {"Z", "X"}
    except (FileNotFoundError, ProcessLookupError):
        return False


@contextlib.contextmanager
def actors(root, mode):
    master = subprocess.Popen(
        [sys.executable, "-I", "-S", str(FIXTURE), "master", str(root), mode],
        start_new_session=True,
    )
    rows = []
    try:
        wait(
            lambda: all(
                (root / f"lane-{i}/roster.json").exists()
                and (root / f"lane-{i}/daemon-ready").exists()
                for i in range(2)
            )
        )
        rows = [
            json.loads((root / f"lane-{i}/roster.json").read_text()) for i in range(2)
        ]
        yield (
            p.observe_process(master.pid),
            [
                p.LaneBinding(
                    p.observe_process(row["supervisor"]),
                    p.observe_process(row["daemon"]) if row["daemon"] else None,
                )
                for row in rows
            ],
        )
    finally:
        if master.poll() is None:
            os.kill(master.pid, signal.SIGSTOP)
        for i, row in enumerate(rows):
            known = [row["supervisor"], row["daemon"]]
            for name in ("births", "ancillary.pid"):
                path = root / f"lane-{i}" / name
                if path.exists():
                    known.extend(int(v) for v in path.read_text().splitlines())
            for pid in set(known) - {None}:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if master.poll() is None:
            master.kill()
        master.wait(timeout=5)


def options(root, controller, lanes):
    @contextlib.contextmanager
    def fence(index):
        with (root / f"lane-{index}/.lane.lock.update.lock").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def population():
        for i in range(2):
            path = root / f"lane-{i}/ancillary.pid"
            if path.exists() and alive(int(path.read_text())):
                raise PopulationPresent("unrecognized_helper_still_present")

    def children(index):
        assert not alive(lanes[index].daemon.pid)
        population()

    def closed():
        population()
        assert all((root / f"lane-{i}/supervisor-cleanup").exists() for i in range(2))

    return dict(
        controller=controller,
        lanes=lanes,
        lane_fence=fence,
        effect_gate=lambda: None,
        population_gate=population,
        population_refusal=PopulationPresent,
        lane_children_gate=children,
        closed_children_gate=closed,
        record_phase=lambda _: None,
        timeout_seconds=0.5,
    )


@pytest.mark.parametrize("mode", ["idle", "idle_all"])
def test_idle_native_wrappers_close_without_fabricating_daemon_birth(tmp_path, monkeypatch, mode):
    with actors(tmp_path, mode) as (controller, original):
        active = [lane for lane in original if lane.daemon is not None]
        idle = [lane.supervisor for lane in original if lane.daemon is None]
        kw = options(tmp_path, controller, original)
        kw.update(lanes=active, idle_supervisors=idle)
        kw["lane_children_gate"] = lambda i: None if original[i].daemon is None else (
            None if not alive(original[i].daemon.pid) else pytest.fail("daemon survives")
        )
        signals = []
        real = signal.pidfd_send_signal
        bound = [controller, *[l.supervisor for l in original], *[l.daemon for l in active]]

        def spy(fd, sig, *args):
            if sig == signal.SIGTERM and signal.SIGTERM not in signals:
                assert all(p.all_threads_stopped(b) for b in bound)
            signals.append(sig)
            return real(fd, sig, *args)

        monkeypatch.setattr(signal, "pidfd_send_signal", spy)
        result = phased.gracefully_close_native_lanes(**kw)
        assert result["closed_lanes"] == [0, 1]
        assert result["controller_exited"] is True
        assert result["callback_settlement_authority"] is False
        assert signals.count(signal.SIGTERM) == len(bound)
        assert signal.SIGKILL not in signals
        assert (tmp_path / "master-cleanup").exists()
        assert all((tmp_path / f"lane-{i}/supervisor-cleanup").exists() for i in range(2))


def test_idle_wrapper_with_unadmitted_live_child_refuses_before_term(tmp_path, monkeypatch):
    with actors(tmp_path, "normal") as (controller, lanes):
        kw = options(tmp_path, controller, lanes)
        kw.update(lanes=lanes[:1], idle_supervisors=[lanes[1].supervisor])
        signals = []
        real = signal.pidfd_send_signal
        monkeypatch.setattr(signal, "pidfd_send_signal", lambda fd, sig, *args: (
            signals.append(sig), real(fd, sig, *args)
        )[1])
        with pytest.raises(p.GracefulRecoveryUnverified, match="idle_supervisor_has_live_child"):
            phased.gracefully_close_native_lanes(**kw)
        assert signal.SIGTERM not in signals
        assert signal.SIGKILL not in signals
        for b in [controller, *[l.supervisor for l in lanes], *[l.daemon for l in lanes]]:
            wait(lambda: p._stat(b.pid)[0] not in {"T", "t"})
            assert alive(b.pid)


@pytest.mark.parametrize("mode", ["normal", "transient", "late_transient"])
def test_all_exact_actors_paused_before_any_term_and_helpers_get_no_signals(
    tmp_path, monkeypatch, mode
):
    with actors(tmp_path, mode) as (controller, lanes):
        kw = options(tmp_path, controller, lanes)
        signals = []
        real = signal.pidfd_send_signal
        all_bindings = [
            controller,
            *[l.supervisor for l in lanes],
            *[l.daemon for l in lanes],
        ]

        def spy(fd, sig, *args):
            if sig == signal.SIGTERM and signal.SIGTERM not in signals:
                assert all(p.all_threads_stopped(b) for b in all_bindings)
            signals.append(sig)
            return real(fd, sig, *args)

        monkeypatch.setattr(signal, "pidfd_send_signal", spy)

        def phase(name):
            if name == "daemon_all_threads_stopped":
                for i in range(2):
                    (tmp_path / f"lane-{i}/helper-trigger").touch()

        kw["record_phase"] = phase
        result = phased.gracefully_close_native_lanes(**kw)
        assert result["controller_exited"] and result["closed_lanes"] == [0, 1]
        assert result["callback_settlement_authority"] is False
        assert signals.count(signal.SIGSTOP) == 5 and signals.count(signal.SIGTERM) == 5
        assert signal.SIGKILL not in signals
        assert (tmp_path / "master-cleanup").exists()
        assert all(
            len((tmp_path / f"lane-{i}/births").read_text().splitlines()) == 1
            for i in range(2)
        )


def test_helper_requiring_parent_progress_times_out_and_resumes_every_actor(
    tmp_path, monkeypatch
):
    with actors(tmp_path, "parent_progress") as (controller, lanes):
        kw = options(tmp_path, controller, lanes)
        signals = []
        real = signal.pidfd_send_signal
        monkeypatch.setattr(
            signal,
            "pidfd_send_signal",
            lambda fd, sig, *a: (signals.append(sig), real(fd, sig, *a))[1],
        )

        def phase(name):
            if name == "daemon_all_threads_stopped":
                for i in range(2):
                    (tmp_path / f"lane-{i}/helper-trigger").touch()

        kw["record_phase"] = phase
        with pytest.raises(PopulationPresent):
            phased.gracefully_close_native_lanes(**kw)
        assert signal.SIGTERM not in signals and signal.SIGKILL not in signals
        for b in [
            controller,
            *[l.supervisor for l in lanes],
            *[l.daemon for l in lanes],
        ]:
            wait(lambda: p._stat(b.pid)[0] not in {"T", "t"})
        assert not (tmp_path / "master-cleanup").exists()


def test_child_born_on_term_blocks_wrapper_and_controller_close(tmp_path, monkeypatch):
    with actors(tmp_path, "ancillary") as (controller, lanes):
        kw = options(tmp_path, controller, lanes)
        signals = []
        real = signal.pidfd_send_signal
        monkeypatch.setattr(
            signal,
            "pidfd_send_signal",
            lambda fd, sig, *a: (signals.append(sig), real(fd, sig, *a))[1],
        )
        with pytest.raises(PopulationPresent):
            phased.gracefully_close_native_lanes(**kw)
        assert signals.count(signal.SIGTERM) == 2
        assert signal.SIGKILL not in signals
        wait(lambda: p._stat(controller.pid)[0] not in {"T", "t"})
        assert not (tmp_path / "master-cleanup").exists()
        for i in range(2):
            assert alive(int((tmp_path / f"lane-{i}/ancillary.pid").read_text()))


@pytest.mark.parametrize(
    "phase",
    [
        "controller_all_threads_stopped",
        "supervisor_all_threads_stopped",
        "daemon_all_threads_stopped",
        "daemon_exit_observed",
        "all_daemons_and_helpers_closed",
    ],
)
def test_journal_or_source_failure_resumes_surviving_exact_actors(tmp_path, phase):
    with actors(tmp_path, "normal") as (controller, lanes):
        kw = options(tmp_path, controller, lanes)

        def fail(name):
            if name == phase:
                raise RuntimeError("source_or_journal_changed")

        kw["record_phase"] = fail
        with pytest.raises(RuntimeError, match="source_or_journal_changed"):
            phased.gracefully_close_native_lanes(**kw)
        for b in [
            controller,
            *[l.supervisor for l in lanes],
            *[l.daemon for l in lanes],
        ]:
            if alive(b.pid):
                wait(lambda: p._stat(b.pid)[0] not in {"T", "t"})
        assert not (tmp_path / "master-cleanup").exists()


def test_actual_source_drift_after_partial_exit_preserves_survivors(tmp_path):
    with actors(tmp_path, "normal") as (controller, lanes):
        kw = options(tmp_path, controller, lanes)
        source = tmp_path / "source-binding"
        source.write_text("old")

        def source_gate():
            if source.read_text() != "old":
                raise RuntimeError("source_changed")

        def phase(name):
            if name == "daemon_exit_observed":
                source.write_text("foreign")

        kw.update(effect_gate=source_gate, record_phase=phase)
        with pytest.raises(RuntimeError, match="source_changed"):
            phased.gracefully_close_native_lanes(**kw)
        assert source.read_text() == "foreign"
        assert not (tmp_path / "master-cleanup").exists()
        for b in [controller, *[l.supervisor for l in lanes]]:
            wait(lambda: p._stat(b.pid)[0] not in {"T", "t"})


def test_fence_release_error_still_resumes_every_suspended_actor(tmp_path):
    with actors(tmp_path, "normal") as (controller, lanes):
        kw = options(tmp_path, controller, lanes)
        native_fence = kw["lane_fence"]

        @contextlib.contextmanager
        def broken_fence(index):
            with native_fence(index):
                yield
            raise OSError("fence_release_unverified")

        def phase(name):
            if name == "daemon_all_threads_stopped":
                raise RuntimeError("original_refusal")

        kw.update(lane_fence=broken_fence, record_phase=phase)
        with pytest.raises(OSError, match="fence_release_unverified"):
            phased.gracefully_close_native_lanes(**kw)
        for b in [
            controller,
            *[l.supervisor for l in lanes],
            *[l.daemon for l in lanes],
        ]:
            wait(lambda: p._stat(b.pid)[0] not in {"T", "t"})
        assert not (tmp_path / "master-cleanup").exists()


def test_wrapper_prepared_journal_failure_precedes_every_wrapper_term(
    tmp_path, monkeypatch
):
    with actors(tmp_path, "normal") as (controller, lanes):
        kw = options(tmp_path, controller, lanes)
        signals = []
        real = signal.pidfd_send_signal

        def spy(fd, sig, *args):
            signals.append(sig)
            return real(fd, sig, *args)

        monkeypatch.setattr(signal, "pidfd_send_signal", spy)

        def phase(name):
            if name == "supervisor_graceful_exit_prepared":
                raise OSError("wrapper_journal_unavailable")

        kw["record_phase"] = phase
        with pytest.raises(OSError, match="wrapper_journal_unavailable"):
            phased.gracefully_close_native_lanes(**kw)
        assert signals.count(signal.SIGTERM) == len(lanes)
        assert signal.SIGKILL not in signals
        for binding in [controller, *[lane.supervisor for lane in lanes]]:
            wait(lambda: p._stat(binding.pid)[0] not in {"T", "t"})
        assert not (tmp_path / "master-cleanup").exists()
        assert not any(
            (tmp_path / f"lane-{i}/supervisor-cleanup").exists() for i in range(2)
        )
